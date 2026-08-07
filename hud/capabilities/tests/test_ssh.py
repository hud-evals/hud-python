from __future__ import annotations

from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock

import asyncssh

from hud.capabilities.base import Capability
from hud.capabilities.ssh import SSHClient

if TYPE_CHECKING:
    import pytest


async def test_connect_keeps_tunneled_connection_active(monkeypatch: pytest.MonkeyPatch) -> None:
    connection = cast("asyncssh.SSHClientConnection", object())
    connect = AsyncMock(return_value=connection)
    monkeypatch.setattr(asyncssh, "connect", connect)

    client = await SSHClient.connect(
        Capability(name="shell", protocol="ssh/2", url="ssh://sandbox.example:8765")
    )

    assert client.conn is connection
    connect.assert_awaited_once_with(
        host="sandbox.example",
        port=8765,
        username="agent",
        client_keys=None,
        known_hosts=None,
        keepalive_interval=15,
        keepalive_count_max=4,
    )
