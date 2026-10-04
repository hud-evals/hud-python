"""Finding a run's capability and opening a client for it."""

from __future__ import annotations

import pytest

from hud.capabilities import Capability, find_capability, open_client

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


async def test_open_client_rejects_a_protocol_no_client_speaks() -> None:
    with pytest.raises(ValueError, match="no client speaks 'unknown/1'"):
        async with open_client(Capability(name="x", protocol="unknown/1", url="tcp://x:1")):
            pass
