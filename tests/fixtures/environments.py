"""Environments built in the test process: ``checked`` (a task the agent passes by
answering ``yes``) and ``workspace_task`` (the task of a real workspace served over SSH).
Environments loaded from a file are in ``tests/fixtures/envs``."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from hud import Environment
from tests.conftest import write_files

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from hud.capabilities import Capability


def checked() -> Environment:
    """The ``checked`` environment: its ``check`` task pays 1.0 when the agent answers ``yes``."""
    env = Environment("checked")

    @env.template()
    async def check(n: int = 0):
        answer = yield "say yes"
        yield 1.0 if answer == "yes" else 0.0

    return env


def workspace_task(
    root: Path,
    *,
    files: Mapping[str, bytes | str] | None = None,
    passes: Callable[[Path], bool] = lambda _root: True,
    capabilities: tuple[Capability, ...] = (),
    settle: float = 0.0,
    **workspace: Any,
):
    """The ``task`` template of an environment serving ``root`` over SSH; ``task.env`` is
    that environment.

    ``files`` seed the workspace before each run; the task scores 1.0 when
    ``passes(root)`` holds ``settle`` seconds after the agent finishes, long
    enough for anything the agent left running to show. ``workspace`` goes to
    :meth:`Environment.workspace` (``env=`` for the shell environment).
    """
    env = Environment("ws", capabilities=capabilities)
    env.workspace(root, guest_path=str(root), **workspace)

    @env.initialize
    async def seed() -> None:
        write_files(root, files or {})

    @env.template()
    async def task(prompt: str = "Do the task."):
        yield prompt
        await asyncio.sleep(settle)
        yield 1.0 if passes(root) else 0.0

    return task
