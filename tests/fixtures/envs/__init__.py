"""Fixture environments: each reward is 1 only when the path it names worked.

Every module is a real environment source, so any placement can serve it: in
this process, in a child process, in a container, or attached by address.
Scenarios load them by path with ``source``. Modules with a ``tasks`` list also
load the way a user's tasks file does, through ``hud eval`` or ``Taskset.from_file``.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent


def source(name: str) -> Path:
    """The path of fixture environment ``name``."""
    path = HERE / f"{name}.py"
    assert path.is_file(), path
    return path
