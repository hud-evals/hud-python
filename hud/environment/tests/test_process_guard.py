"""Process guard connection policy."""

from __future__ import annotations

import socket
import struct

import pytest

from hud.environment.process_guard import _destination, _refused

_RELAY = ("127.0.0.2", 80)


def _sockaddr(host: str, port: int) -> bytes:
    if ":" in host:
        address = socket.inet_pton(socket.AF_INET6, host)
        return (
            struct.pack("H", socket.AF_INET6)
            + struct.pack("!H", port)
            + bytes(4)
            + address
            + bytes(4)
        )
    address = socket.inet_pton(socket.AF_INET, host)
    return struct.pack("H", socket.AF_INET) + struct.pack("!H", port) + address + bytes(8)


@pytest.mark.parametrize(
    ("host", "port", "refused"),
    [
        ("127.0.0.2", 80, True),
        ("::ffff:127.0.0.2", 80, True),
        ("0.0.0.0", 80, True),
        ("::ffff:0.0.0.0", 80, True),
        ("::", 80, True),
        ("127.0.0.1", 80, False),
        ("0.0.0.0", 3000, False),
    ],
    ids=[
        "relay",
        "mapped",
        "unspecified",
        "mapped-unspecified",
        "v6-unspecified",
        "other-host",
        "other-port",
    ],
)
def test_unbound_process_cannot_reach_a_protected_port(host: str, port: int, refused: bool) -> None:
    assert _refused(_destination(_sockaddr(host, port)), False, {_RELAY}, set()) is refused


def test_bound_process_reaches_its_connection_through_an_unspecified_address() -> None:
    assert not _refused(_destination(_sockaddr("0.0.0.0", 80)), True, {_RELAY}, {_RELAY})


def test_unspecified_address_is_refused_while_another_connection_shares_the_port() -> None:
    other = ("127.0.0.1", 80)
    assert _refused(_destination(_sockaddr("0.0.0.0", 80)), True, {_RELAY, other}, {_RELAY})
