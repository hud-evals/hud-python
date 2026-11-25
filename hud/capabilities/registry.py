"""Opening a capability with the client that speaks its protocol."""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, TypeVar

from .base import CapabilityClient
from .mcp import MCPClient
from .rfb import RFBClient
from .ssh import SSHClient

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from .base import Capability

_CONNECT_ATTEMPTS = 3
_CONNECT_BASE_DELAY_SECONDS = 0.25
_CLIENTS: dict[str, type[CapabilityClient]] = {}
ClientT = TypeVar("ClientT", bound=type[CapabilityClient])


def register_client(cls: ClientT) -> ClientT:
    """Make ``open_client`` use ``cls`` for capabilities whose protocol is ``cls.protocol``.

    Returns ``cls``, so it also works as a class decorator. Raises ``ValueError`` when
    another class is already registered for the protocol.
    """
    registered = _CLIENTS.setdefault(cls.protocol, cls)
    if registered is not cls:
        raise ValueError(
            f"protocol {cls.protocol!r} is already registered to "
            f"{registered.__module__}.{registered.__qualname__}; "
            f"cannot register {cls.__module__}.{cls.__qualname__}"
        )
    return cls


for _client in (SSHClient, RFBClient, MCPClient):
    register_client(_client)


@asynccontextmanager
async def open_client(capability: Capability) -> AsyncGenerator[CapabilityClient]:
    """A connected client for ``capability``'s protocol, closed on exit.

    Connecting retries while the capability's server comes up.
    """
    client_cls = _CLIENTS.get(capability.protocol)
    if client_cls is None:
        raise ValueError(
            f"no client speaks {capability.protocol!r}; register one with register_client"
        )
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
