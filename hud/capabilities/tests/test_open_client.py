"""Finding a run's capability and opening a client for it."""

from __future__ import annotations

from typing import ClassVar

import pytest

import hud.capabilities as capabilities_module
from hud.capabilities import Capability, CapabilityClient, find_capability, open_client

_ECHO = Capability(name="echo", protocol="test/1", url="tcp://environment:1")


def test_find_capability_matches_name_protocol_or_family() -> None:
    shell = Capability.ssh(url="ssh://box:22", host_pubkey="key")
    capabilities = [_ECHO, shell]

    assert find_capability(capabilities, "echo") is _ECHO
    assert find_capability(capabilities, "ssh/2") is shell
    assert find_capability(capabilities, "ssh") is shell
    with pytest.raises(KeyError, match="no capability 'cdp'"):
        find_capability(capabilities, "cdp")
    with pytest.raises(KeyError, match="ambiguous"):
        find_capability(
            [_ECHO, Capability(name="other", protocol="test/1", url="tcp://x:1")], "test"
        )


async def test_open_client_retries_while_the_server_comes_up(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RetryingClient(CapabilityClient):
        protocol: ClassVar[str] = "test/1"
        attempts = 0

        @classmethod
        async def connect(cls, cap: Capability) -> RetryingClient:
            del cap
            cls.attempts += 1
            if cls.attempts < 3:
                raise ConnectionError("connection reset")
            return cls()

        async def close(self) -> None:
            pass

    monkeypatch.setitem(capabilities_module._CLIENTS, RetryingClient.protocol, RetryingClient)
    monkeypatch.setattr(capabilities_module, "_CONNECT_BASE_DELAY_SECONDS", 0)

    async with open_client(_ECHO) as client:
        assert isinstance(client, RetryingClient)

    assert RetryingClient.attempts == 3


async def test_open_client_closes_on_exit_even_when_closing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    closed: list[str] = []

    class BrokenCloseClient(CapabilityClient):
        protocol: ClassVar[str] = "test/1"

        @classmethod
        async def connect(cls, cap: Capability) -> BrokenCloseClient:
            del cap
            return cls()

        async def close(self) -> None:
            closed.append("echo")
            raise RuntimeError("transport already gone")

    monkeypatch.setitem(capabilities_module._CLIENTS, BrokenCloseClient.protocol, BrokenCloseClient)

    async with open_client(_ECHO):
        pass

    assert closed == ["echo"]


async def test_open_client_rejects_a_protocol_no_client_speaks() -> None:
    with pytest.raises(ValueError, match="no client speaks 'unknown/1'"):
        async with open_client(Capability(name="x", protocol="unknown/1", url="tcp://x:1")):
            pass
