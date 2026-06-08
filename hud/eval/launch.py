"""launch: connect a ``HudClient`` to a spun-up ``Sandbox``.

A client-side convenience on top of the (decoupled) sandbox layer: ``launch``
brings up a sandbox and attaches a client to its runtime, tearing both down on
exit. ``Variant`` (see :mod:`hud.eval.variant`) sits on top of this.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from hud.client import HudClient

from .sandbox import as_sandbox

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from hud.environment import Environment

    from .sandbox import Sandbox


async def _connect_ready(
    host: str,
    port: int,
    *,
    auth_token: str | None = None,
    ready_timeout: float = 120.0,
    interval: float = 0.5,
) -> HudClient:
    """Connect to a control channel, retrying until it accepts or ``ready_timeout``.

    A freshly-spun sandbox may not be serving yet; the client owns waiting for
    readiness by retrying the connect (the sandbox just hands back a url).
    """
    loop = asyncio.get_event_loop()
    deadline = loop.time() + ready_timeout
    while True:
        try:
            return await HudClient.connect(host, port, auth_token=auth_token)
        except OSError:
            if loop.time() >= deadline:
                raise
            await asyncio.sleep(interval)


@asynccontextmanager
async def launch(ref: Sandbox | Environment) -> AsyncIterator[HudClient]:
    """Bring up a substrate for ``ref``, attach a client, tear it down on exit.

    ``ref`` is a :class:`~hud.eval.sandbox.Sandbox` (local, container, HUD-hosted, …)
    or a live ``Environment`` (wrapped in a ``LocalSandbox``). ``launch`` *owns* what
    it spins up; the client connects to the sandbox's runtime url, retrying until the
    control channel is ready.
    """
    sandbox = as_sandbox(ref)
    async with sandbox as runtime:
        parts = urlsplit(runtime.url)
        if parts.scheme not in ("", "tcp"):
            raise NotImplementedError(
                f"control transport {parts.scheme!r} not supported yet (only tcp://)",
            )
        client = await _connect_ready(
            parts.hostname or "127.0.0.1",
            parts.port or 0,
            auth_token=runtime.auth_token,
        )
        async with client:
            yield client


__all__ = ["launch"]
