"""Load authored Python source files and directories.

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


def _module_name(file: Path) -> tuple[str, Path]:
    parts = [] if file.name == "__init__.py" else [file.stem]
    directory = file.parent
    while (directory / "__init__.py").is_file():
        parts.insert(0, directory.name)
        directory = directory.parent
    if file.parent != directory:
        package_root = directory / parts[0]
        for name, module in tuple(sys.modules.items()):
            if name != parts[0] and not name.startswith(f"{parts[0]}."):
                continue
            locations = (*getattr(module, "__path__", ()), getattr(module, "__file__", None))
            if any(
                location and not Path(location).resolve().is_relative_to(package_root)
                for location in locations
            ):
                raise ValueError(
                    f"cannot load {file}: package {parts[0]!r} is already imported from "
                    "a different source root; load this source in a separate process"
                )
    else:
        module = sys.modules.get(file.stem)
        if module is not None and getattr(module, "__file__", None) != str(file):
            return f"_hud_mod_{file.stem}_{abs(hash(str(file)))}", directory
    return ".".join(parts), directory


def load_module(path: str | Path) -> ModuleType:
    """Import a Python file as a throwaway module and return it.

    The import root is on ``sys.path`` during import. Packages keep their
    qualified names; standalone files use a private name when theirs is taken.
    The source module entry is restored afterward; imported dependencies use
    normal Python caching.
    """
    file = Path(path).resolve()
    if not file.is_file():
        raise FileNotFoundError(f"module not found: {path}")

    mod_name, import_root = _module_name(file)
    spec = importlib.util.spec_from_file_location(mod_name, file)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import module: {file}")

    parent = str(import_root)
    sys.path.insert(0, parent)
    try:
        package = mod_name.rpartition(".")[0]
        if package:
            importlib.import_module(package)
            imported = sys.modules.get(mod_name)
            if imported is not None:
                return imported
        previous = sys.modules.get(mod_name)
        module = importlib.util.module_from_spec(spec)
        sys.modules[mod_name] = module
        try:
            spec.loader.exec_module(module)
            return module
        finally:
            if previous is None:
                sys.modules.pop(mod_name, None)
            else:
                sys.modules[mod_name] = previous
    finally:
        with contextlib.suppress(ValueError):
            sys.path.remove(parent)


def iter_modules(path: str | Path) -> Iterator[ModuleType]:
    """Import a ``.py`` file, or every ``.py`` in a directory, yielding modules.

    A file import fails loudly. Directory scans skip packaging/test scaffolding
    and files that fail to import (a source dir may contain unrelated files).
    Directory imports use Python's module cache.
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
        name, _ = _module_name(file)
        module = sys.modules.get(name)
        if module is None or getattr(module, "__file__", None) != str(file):
            try:
                module = load_module(file)
            except ImportError:
                LOGGER.debug("skipping %s (failed to import)", file.name)
                continue
            sys.modules[name] = module
        yield module


__all__ = ["iter_modules", "load_module"]
