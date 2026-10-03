"""Capability declarations + clients."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .base import Capability, CapabilityClient, find_capability
from .cdp import CDPClient
from .connection import Connection
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from .robot import RobotClient


_CONNECT_ATTEMPTS = 3
_CONNECT_BASE_DELAY_SECONDS = 0.25
_CLIENTS: dict[str, type[CapabilityClient]] = {
    cls.protocol: cls for cls in (SSHClient, RFBClient, MCPClient, CDPClient)
}


def __getattr__(name: str) -> object:
    # RobotClient needs the robot extra, so it is imported on first access.
    if name == "RobotClient":
        from .robot import RobotClient

        return RobotClient
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


@asynccontextmanager
async def open_client(capability: Capability) -> AsyncIterator[CapabilityClient]:
    """A connected client for ``capability``'s protocol, closed on exit.

    Connecting retries while the capability's server comes up.
    """
    client_cls = _CLIENTS.get(capability.protocol)
    if client_cls is None and capability.protocol.split("/", 1)[0] == "openpi":
        from .robot import RobotClient

        client_cls = _CLIENTS[RobotClient.protocol] = RobotClient
    if client_cls is None:
        raise ValueError(f"no client speaks {capability.protocol!r}")
    for attempt in range(_CONNECT_ATTEMPTS - 1):
        try:
            client = await client_cls.connect(capability)
            break
        except ConnectionError:
            await asyncio.sleep(_CONNECT_BASE_DELAY_SECONDS * 2**attempt)
    else:
        client = await client_cls.connect(capability)
    try:
        yield client
    finally:
        with contextlib.suppress(Exception):
            await client.close()


__all__ = [
    "CDPClient",
    "Capability",
    "CapabilityClient",
    "Connection",
    "MCPClient",
    "RFBClient",
    "RobotClient",
    "SSHClient",
    "find_capability",
    "open_client",
]
