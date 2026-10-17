"""The internet, as a process in the test reaches it: by name, through DNS.

:func:`internet` stands in for it for the rest of a test. Each name in
``servers`` resolves to a public address of its own, and a connection to that
address reaches the local port the test serves the host on, so code that
refuses anything but the public internet still reaches a test server. A name in
``resolve`` resolves to the address given (``"127.0.0.1"`` for a name that
points back into the substrate). Every other name fails to resolve, as it would
offline. IP literals and ``localhost`` resolve as usual, so loopback traffic in
the test process is unaffected.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest


def internet(
    monkeypatch: pytest.MonkeyPatch,
    servers: Mapping[str, int] | None = None,
    resolve: Mapping[str, str] | None = None,
) -> None:
    """Serve each of ``servers`` (name to local port) as a host on the public internet.

    In force until the test ends.
    """
    public = {name: f"6.0.0.{index}" for index, name in enumerate(servers or {}, start=1)}
    ports = {public[name]: port for name, port in (servers or {}).items()}
    addresses = {**public, **(resolve or {})}
    getaddrinfo = socket.getaddrinfo
    connect = socket.socket.connect

    def lookup(host: Any, port: Any, *args: Any, **kwargs: Any) -> Any:
        name = host.decode() if isinstance(host, bytes) else host
        if name in addresses:
            return getaddrinfo(addresses[name], port, *args, **kwargs)
        try:
            ipaddress.ip_address(name)
        except ValueError:
            if name not in (None, "localhost"):
                raise socket.gaierror(socket.EAI_NONAME, f"{name} does not resolve") from None
        return getaddrinfo(host, port, *args, **kwargs)

    def dial(self: socket.socket, address: Any) -> None:
        if isinstance(address, tuple) and address[0] in ports:
            address = ("127.0.0.1", ports[address[0]])
        connect(self, address)

    monkeypatch.setattr(socket, "getaddrinfo", lookup)
    monkeypatch.setattr(socket.socket, "connect", dial)
