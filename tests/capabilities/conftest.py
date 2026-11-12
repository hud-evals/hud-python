"""A capability binding reached through a relay that can drop it."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import replace
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

from tests.fixtures.relay import relay

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from hud.capabilities import Capability
    from tests.fixtures.relay import FlakyRelay


@asynccontextmanager
async def relayed(binding: Capability) -> AsyncGenerator[tuple[Capability, FlakyRelay]]:
    """``binding``, reached through a relay that can sever, refuse, and stall its connections."""
    parts = urlsplit(binding.url)
    assert parts.hostname is not None and parts.port is not None
    async with relay(parts.hostname, parts.port) as flaky:
        user = f"{parts.username}@" if parts.username else ""
        netloc = f"{user}127.0.0.1:{flaky.port}"
        url = urlunsplit((parts.scheme, netloc, parts.path, parts.query, ""))
        yield replace(binding, url=url), flaky
