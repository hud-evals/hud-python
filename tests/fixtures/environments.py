"""Environments built in the test process: ``checked`` (a task the agent passes by
answering ``yes``) and ``workspace_task`` (the task of a real workspace served over SSH).
Environments loaded from a file are in ``tests/fixtures/envs``."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from hud import Environment

if TYPE_CHECKING:
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


def workspace_task(root: Path, *, capabilities: tuple[Capability, ...] = (), **workspace: Any):
    """The ``task`` template of an environment serving ``root`` over SSH; ``task.env`` is
    that environment.

    The task scores 1.0. ``workspace`` goes to ``Environment.workspace`` (``env=`` for the
    shell environment).
    """
    env = Environment("ws", capabilities=capabilities)
    env.workspace(root, guest_path=str(root), **workspace)

    @env.template()
    async def task(prompt: str = "Do the task."):
        yield prompt
        yield 1.0

    return task
