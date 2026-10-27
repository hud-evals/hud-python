"""A served workspace, reached the way an agent reaches it: over its ``ssh/2`` binding.

``shell`` serves ``env.workspace(root, ...)`` on loopback and yields the
:class:`~hud.environment.Workspace` it attached (for what the environment itself
does with it: verifier runs, launches) with an SSH client on its published binding.

Rows marked ``SANDBOX`` need Linux, root and bubblewrap; they run in the sandbox
lane (``-m "e2e and sandbox"``), where every session runs in the bubblewrap
sandbox. Unmarked rows run wherever the suite runs, sandboxed or not.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, cast

import pytest

from hud.clients import connect
from hud.environment import Environment
from hud.eval import LocalRuntime, Task

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from hud.capabilities import SSHClient
    from hud.environment import Workspace

SANDBOX = (pytest.mark.e2e, pytest.mark.sandbox)

if sys.platform == "win32":
    collect_ignore_glob = ["test_*.py"]


@asynccontextmanager
async def shell(root: Path, **options: Any) -> AsyncIterator[tuple[Workspace, SSHClient]]:
    """Serve a workspace over ``root`` and connect to its shell binding."""
    env = Environment("workspace")
    workspace = env.workspace(root, track_files=False, **options)
    async with (
        LocalRuntime(env)(Task(env=env.name, id="scenario")) as runtime,
        connect(runtime) as client,
    ):
        yield workspace, cast("SSHClient", await client.open("shell"))
