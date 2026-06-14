"""Env-run daemons publish capabilities at serve time, never at declaration.

Publication is protocol-agnostic: a capability backer is started from an
``@env.initialize`` hook and published with ``env.add_capability(...)``, deferring
everything — keys, sockets — until the env actually serves. The manifest carries
the published address, and ``env.stop()`` runs the matching shutdown hooks.
"""

from __future__ import annotations

from typing import cast

import pytest

from hud.capabilities import Capability
from hud.environment import Environment

from .conftest import served


async def test_any_initialize_hook_can_publish_a_capability() -> None:
    """Publication is protocol-agnostic: no SDK type per daemon kind."""
    import asyncio

    server = await asyncio.start_server(lambda r, w: w.close(), "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    torn_down = False

    env = Environment("browser-env")

    @env.initialize
    async def _up() -> None:
        env.add_capability(Capability.cdp(name="browser", url=f"ws://127.0.0.1:{port}"))

    @env.shutdown
    async def _down() -> None:
        nonlocal torn_down
        torn_down = True
        server.close()

    assert env.capabilities == []
    async with served(env) as client:
        cap = client.binding("browser")
        assert cap.protocol == "cdp/1.3"
        assert cap.url.startswith("ws://127.0.0.1:")
    assert torn_down  # env.stop() ran the shutdown hook


async def test_loopback_declarations_are_forwarded_and_remote_ones_pass_through() -> None:
    local = Capability.cdp(name="browser", url="ws://127.0.0.1:9222")
    remote = Capability.ssh(name="box", url="ssh://box.example.com:22", host_pubkey="ssh-ed25519 x")
    env = Environment("mixed-env", capabilities=[local, remote])

    async with served(env) as client:
        forwarded = client.binding("browser")
        # Loopback means substrate-local: the client substitutes a local
        # forwarder address, everything else about the capability intact.
        assert forwarded.url != local.url
        assert forwarded.url.startswith("ws://127.0.0.1:")
        assert (forwarded.name, forwarded.protocol) == (local.name, local.protocol)
        # Globally-reachable addresses are the client's to dial directly.
        assert client.binding("box") == remote


def test_a_capability_without_a_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="initialize hook"):
        Environment("bad", capabilities=[Capability(name="b", protocol="cdp/1.3", url="")])


def test_non_capability_entries_are_rejected() -> None:
    with pytest.raises(TypeError, match="expected Capability"):
        Environment("bad", capabilities=cast("list[Capability]", [object()]))
