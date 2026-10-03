"""Opening a capability with the client that speaks its protocol."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from .cdp import CDPClient
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from .base import Capability, CapabilityClient

_CONNECT_ATTEMPTS = 3
_CONNECT_BASE_DELAY_SECONDS = 0.25
_CLIENTS: dict[str, type[CapabilityClient]] = {
    cls.protocol: cls for cls in (SSHClient, RFBClient, MCPClient, CDPClient)
}


@asynccontextmanager
async def open_client(capability: Capability) -> AsyncGenerator[CapabilityClient]:
    """A connected client for ``capability``'s protocol, closed on exit.

    Connecting retries while the capability's server comes up.
    """
    client_cls = _CLIENTS.get(capability.protocol)
    if client_cls is None and capability.protocol.split("/", 1)[0] == "openpi":
        # RobotClient needs the robot extra, so it joins the registry on first open.
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
