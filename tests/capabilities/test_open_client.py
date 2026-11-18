"""Finding a run's capability and opening a client for it."""

from __future__ import annotations

from typing import ClassVar, Self

import pytest

from hud.capabilities import (
    Capability,
    CapabilityClient,
    find_capability,
    open_client,
    register_client,
)

_ECHO = Capability(name="echo", protocol="test/1", url="tcp://environment:1")


@register_client
class EchoClient(CapabilityClient):
    """A client for a protocol the SDK does not know, registered the way a harness package would."""

    protocol: ClassVar[str] = "echo/1"

    def __init__(self, capability: Capability) -> None:
        self.capability = capability
        self.closed = False

    @classmethod
    async def connect(cls, cap: Capability) -> Self:
        return cls(cap)

    async def close(self) -> None:
        self.closed = True


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


async def test_open_client_rejects_a_protocol_no_client_speaks() -> None:
    with pytest.raises(ValueError, match="no client speaks 'unknown/1'"):
        async with open_client(Capability(name="x", protocol="unknown/1", url="tcp://x:1")):
            pass


async def test_a_registered_client_class_opens_its_protocols_capabilities() -> None:
    echo = Capability(name="echo", protocol="echo/1", url="tcp://environment:7100")
    async with open_client(echo) as opened:
        assert isinstance(opened, EchoClient)
        assert not opened.closed

    assert (opened.capability.name, opened.closed) == ("echo", True)


def test_a_second_client_class_for_a_registered_protocol_is_refused() -> None:
    class OtherEchoClient(EchoClient):
        pass

    with pytest.raises(ValueError, match=r"'echo/1' is already registered to .*\.EchoClient"):
        register_client(OtherEchoClient)
    assert register_client(EchoClient) is EchoClient
