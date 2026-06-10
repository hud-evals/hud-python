"""Harnesses for protocol-level environment tests.

Inline-defined envs have no source file to ``spawn``, so :func:`served` drives
the connect path against a loopback substrate served by this process (the
same ``_local`` serving ``AgentTool`` adapts inside a placed substrate). This
is a test harness, not an engine placement.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from hud.clients import connect
from hud.environment.runtime import _local

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.clients import HudClient
    from hud.environment import Environment


@asynccontextmanager
async def served(env: Environment) -> AsyncIterator[HudClient]:
    """Serve *env* on a loopback substrate and yield a connected client."""
    async with _local(env) as runtime, connect(runtime) as client:
        yield client
